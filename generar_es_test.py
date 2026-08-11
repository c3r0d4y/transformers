#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Genera data/es_test.csv: 100 correos sinteticos en espanol (50 phishing, 50 legitimos).

Todos los textos son inventados. Las entidades y dominios son ficticios
(.example es un TLD reservado, no resoluble) para que el archivo sirva como
banco de prueba de deteccion sin ser una plantilla utilizable contra nadie.
"""
import sys
import pandas as pd

PHISHING = [
    "Estimado cliente, hemos detectado un acceso no autorizado a su cuenta de Banco Meridional. Por su seguridad la hemos bloqueado temporalmente. Verifique su identidad en las próximas 24 horas en http://meridional-verificacion.example/acceso o su cuenta será cancelada de forma permanente.",
    "Su paquete no pudo ser entregado por un dato incompleto en la dirección. Pague la tarifa de reenvío de $48.00 antes de las 18:00 de hoy en http://paqueteria-reenvio.example/pago para programar un nuevo intento de entrega.",
    "Adjunto encontrará la factura pendiente FAC-90231 por $12,450.00. Si el pago no se recibe en 48 horas se aplicarán recargos moratorios. Descargue el documento aquí: http://facturas-pendientes.example/FAC90231.zip",
    "Recibimos una solicitud para restablecer la contraseña de su cuenta. Si no fue usted, confirme su identidad de inmediato en http://cuenta-segura-restablecer.example ingresando su usuario y su contraseña actual.",
    "La Autoridad Fiscal Nacional le informa que tiene un saldo a favor de $8,320.00 pendiente de devolución. Complete el formulario con sus datos bancarios en http://devolucion-fiscal.example antes del cierre del ejercicio.",
    "Necesito que realices una transferencia urgente de $85,000 a un proveedor nuevo antes de las dos de la tarde. Estoy en junta y no puedo contestar llamadas. Responde a este correo y te envío los datos de la cuenta. No lo comentes con nadie del equipo todavía.",
    "Su buzón de correo ha alcanzado el 98% de su capacidad y los mensajes entrantes serán rechazados. Amplíe su almacenamiento sin costo validando su cuenta en http://webmail-ampliar.example/cuota",
    "Buen día, vimos su perfil y queremos ofrecerle un puesto de asistente administrativo remoto con sueldo de $28,000 mensuales por tres horas al día. No requiere experiencia. Envíe copia de su identificación y su número de cuenta para iniciar el proceso.",
    "Su suscripción a StreamMax será cancelada porque no pudimos procesar el cobro de este mes. Actualice su método de pago en http://streammax-facturacion.example/actualizar para no perder su historial.",
    "Felicidades, su número resultó ganador de $150,000 en el sorteo anual. Para liberar el premio únicamente debe cubrir el impuesto de traslado de $2,500. Responda con su nombre completo, su identificación oficial y su número de cuenta.",
    "Inversión garantizada en criptomonedas con rendimiento del 12% semanal. El cupo está limitado a veinte personas. Regístrese hoy en http://cripto-rendimiento.example y duplique su capital en un mes.",
    "Su cuenta será desactivada por incumplir nuestras normas comunitarias. Si considera que se trata de un error, presente su apelación en http://apelacion-cuenta.example/formulario dentro de las próximas 12 horas.",
    "Alerta de seguridad: se detectaron tres amenazas activas en su equipo. Descargue el limpiador oficial desde http://limpieza-sistema.example/instalar.exe y ejecútelo como administrador para eliminarlas.",
    "Se le notifica que existe un requerimiento judicial en su contra por un adeudo no cubierto. Consulte el expediente digital en http://notificacion-judicial.example/expediente antes de que se giren órdenes de embargo.",
    "Ayúdanos a llevar despensas a las familias afectadas por las inundaciones, cualquier donativo cuenta. Deposita a la cuenta 4152 3130 8827 6644 a nombre de Fundación Manos Unidas y envíanos tu comprobante por este medio.",
    "Recursos Humanos migró el sistema de nómina. Es obligatorio reconfirmar tu cuenta bancaria antes del viernes para que tu pago no se retrase. Actualiza tus datos en http://nomina-empleados.example/alta",
    "Un compañero compartió contigo el documento Presupuesto_2026_confidencial.xlsx. Inicia sesión con tu correo institucional para poder verlo: http://documentos-compartidos.example/abrir",
    "Tus credenciales de VPN expiran hoy. Renuévalas en http://vpn-renovacion.example/login para no perder el acceso remoto. Si no lo haces antes de la medianoche tendrás que acudir presencialmente a soporte.",
    "Aviso de corte: su servicio de energía será suspendido en 24 horas por un adeudo de $1,890.00. Regularice su pago en http://pagos-servicio-luz.example y evite el cargo por reconexión.",
    "Detectamos un inicio de sesión desde Kiev, Ucrania, en su cuenta. Si no fue usted, cambie su contraseña inmediatamente aquí: http://seguridad-sesiones.example/bloquear",
    "Hola, te escribo porque vi tu foto y me pareciste muy interesante. Me gustaría conocerte mejor, tengo fotos privadas para compartirte. Escríbeme por aquí: http://conocernos-hoy.example/perfil",
    "Le informamos que un familiar lejano falleció en el extranjero dejando una herencia de 4.5 millones de dólares sin reclamar. Como albacea necesito sus datos personales y un anticipo para gastos notariales para iniciar el trámite.",
    "Su tarjeta con terminación 4471 fue bloqueada por compras sospechosas de $9,400 en el extranjero. Si no reconoce el cargo, cancele la operación en http://tarjeta-bloqueo.example/cancelar confirmando su NIP.",
    "Actualización obligatoria de su módem. Ingrese a http://config-modem.example con el usuario y la contraseña que aparecen en la etiqueta del equipo para aplicar el parche de seguridad.",
    "Fuiste preseleccionado para la beca de excelencia por $45,000 semestrales. Confirma tu lugar subiendo tu identificación, tu comprobante de domicilio y tu estado de cuenta en http://becas-registro.example/confirmar",
    "El portal de calificaciones cambió de dominio. Consulta tus resultados finales en http://portal-calificaciones.example accediendo con tu matrícula y la contraseña de siempre.",
    "Su reembolso de $1,299.00 ya está listo pero no pudimos depositarlo. Ingrese los datos de su tarjeta en http://reembolso-pendiente.example/cobrar para recibirlo en 24 horas.",
    "Se renovó automáticamente su licencia anual por $4,999.00. Si no autorizó este cargo, cancele la suscripción llamando al 55 8812 4490 o desde http://cancelar-cargo.example",
    "Su espacio en la nube está lleno y sus archivos serán eliminados en cinco días. Confirme su cuenta en http://nube-almacenamiento.example/verificar para conservarlos.",
    "Estamos validando su identidad. En un momento recibirá un código de seis dígitos por mensaje de texto. Por favor responda a este correo con ese código para completar la verificación de seguridad.",
    "Soporte técnico: migraremos todos los buzones esta noche. Para no perder sus correos, inicie sesión en el nuevo servidor http://migracion-buzones.example y confirme su contraseña actual.",
    "Te comparto la grabación de la junta de ayer, revísala antes de mañana por favor. Se abre con tu cuenta de correo: http://grabacion-junta.example/ver",
    "Estoy ocupado en una reunión con clientes y no puedo salir. Necesito que compres cinco tarjetas de regalo de $2,000 cada una y me envíes las fotos de los códigos. Te lo reembolso hoy mismo con el gasto de la tarjeta corporativa.",
    "Buenas tardes, le informamos que cambiamos de institución bancaria. A partir de hoy los pagos deben depositarse a la cuenta 002180041234567890 a nombre de Suministros Delta. Favor de actualizar el pago pendiente de esta semana.",
    "Su siniestro fue aprobado por $32,000. Para liberar el pago necesitamos que confirme sus datos bancarios y suba su identificación en http://siniestro-pago.example/liberar",
    "Sus resultados de laboratorio ya están disponibles. Consúltelos en http://resultados-clinica.example ingresando su fecha de nacimiento y el número de la tarjeta con la que pagó el estudio.",
    "Usted tiene un crédito preaprobado de $250,000 sin consultar buró y sin aval. Solo cubra el seguro de apertura de $3,200 y el depósito se realiza el mismo día. Responda este correo para apartar su folio.",
    "Su envío internacional está retenido en aduana. Debe cubrir $890.00 por concepto de impuestos de importación en http://aduana-liberacion.example/pagar antes de 72 horas o el paquete será devuelto al remitente.",
    "Tiene un documento pendiente de firma electrónica: Contrato_Servicios_2026.pdf. Fírmelo aquí: http://firma-digital.example/documento. El enlace expira en seis horas.",
    "Conteste esta encuesta de tres minutos y reciba $500 en una tarjeta de regalo. Solo debe registrar sus datos y una tarjeta válida para verificar que no es un robot: http://encuesta-premiada.example",
    "Su monedero digital tiene $2,780.00 sin reclamar de un reembolso cancelado. Vincule su tarjeta en http://monedero-reclamo.example/vincular para recibir el saldo antes de que se libere.",
    "Se confirmó su pedido de una computadora portátil por $34,999.00 con envío a Monterrey. Si usted no realizó esta compra, cancélela de inmediato en http://cancelar-pedido.example/orden88213",
    "Sus 12,400 puntos de recompensa vencen esta semana. Canjéalos por saldo en http://puntos-recompensa.example accediendo con el número de tu línea y tu NIP telefónico.",
    "Programa de apoyo social: usted califica para recibir $6,000 bimestrales. Registre su solicitud en http://apoyo-bienestar.example/registro con su clave única de población, su clave de elector y su número de cuenta.",
    "El departamento está disponible desde el próximo mes y hay varios interesados. Como estoy fuera del país, aparte el inmueble depositando un mes de renta a la cuenta que le indico y le envío las llaves por paquetería.",
    "Felicidades, tu solicitud de prácticas profesionales fue aceptada. Para generar tu carta de aceptación realiza el pago de $850 por gastos administrativos en http://practicas-aceptacion.example/pago",
    "Su registro al Congreso Internacional de Tecnología está incompleto. Complete el pago de $1,200 dólares en http://congreso-registro.example antes del día 15 o su ponencia será retirada del programa.",
    "La licencia de su software de diseño expiró. Descargue el activador desde http://licencia-activador.example/setup.exe y desactive temporalmente su antivirus para que la instalación se complete correctamente.",
    "Hola, soy Andrea del área de finanzas. Necesito el listado de empleados con su registro fiscal y su fecha de nacimiento para un trámite urgente ante la aseguradora. Envíamelo hoy mismo por favor y no lo comentes con nadie más.",
    "Le llamamos del departamento antifraude de su banco. Para revertir los cargos no reconocidos, siga las instrucciones y proporcione el código de seguridad que le enviamos por mensaje. No cuelgue ni consulte su aplicación durante el proceso.",
]

LEGITIMOS = [
    "Gracias por tu compra. Tu pedido 4471-A quedó confirmado y se entregará entre el 12 y el 14 de agosto. Puedes ver el detalle en http://tienda-lumen.example/mis-pedidos/4471A",
    "Boletín mensual: en esta edición hablamos de tres técnicas de regularización en redes neuronales y reseñamos dos artículos recientes. Si ya no quieres recibirlo puedes darte de baja al final del correo.",
    "Convoco a la reunión de seguimiento del proyecto el jueves a las 10:00 en la sala 3. La agenda es avances del sprint, riesgos y fecha de entrega. Si alguien no puede asistir avísenme antes del miércoles.",
    "El respaldo automático del servidor de archivos terminó correctamente a las 03:15. Se copiaron 412 gigabytes sin errores. Este es un mensaje automático del sistema, no es necesario responder.",
    "Tu ticket SOP-2291 fue actualizado. Revisamos el problema de impresión y reinstalamos el controlador. Avísanos si vuelve a ocurrir y reabrimos el caso.",
    "Te recordamos tu cita de limpieza dental el martes 18 a las 16:30. Si necesitas reprogramar, responde a este mensaje o llámanos con al menos 24 horas de anticipación.",
    "Hola Luis, disculpa la insistencia. ¿Me puedes pasar el documento de requerimientos que revisamos la semana pasada? Lo necesito para cerrar el reporte del viernes. Gracias.",
    "Recibimos tu solicitud para restablecer la contraseña. Puedes crear una nueva desde la aplicación, en Ajustes, Seguridad, Cambiar contraseña. Nunca te pediremos tu contraseña por correo ni por teléfono.",
    "Tu paquete va en camino. La guía es 7742119083 y la entrega estimada es el jueves. Puedes seguir el envío en http://paqueteria-andina.example/rastreo/7742119083",
    "Tu estado de cuenta del periodo de julio ya está disponible. Consúltalo desde la aplicación o entrando directamente al sitio del banco. Por seguridad, este aviso no incluye ningún enlace.",
    "Convocatoria abierta: el congreso recibe artículos hasta el 30 de septiembre. Los trabajos deben enviarse en formato de dos columnas y no exceder ocho páginas. Las bases completas están en http://congreso-cic.example/convocatoria",
    "Aviso académico: la clase de Aprendizaje Profundo del miércoles se recorre al viernes a la misma hora por la jornada de mantenimiento del laboratorio de cómputo.",
    "Revisé tu capítulo 3. Va bien encaminado, pero la sección de metodología necesita más detalle sobre cómo partiste los datos. Te dejé comentarios en el documento, lo platicamos el lunes.",
    "Buen día, adjuntamos la factura correspondiente al servicio de mantenimiento de julio, conforme al contrato vigente. Cualquier aclaración con gusto la atendemos en el teléfono de siempre.",
    "El equipo va a comer el viernes para festejar el cierre del proyecto. Es a la una en el restaurante de la esquina. Confirmen para reservar la mesa.",
    "Del 1 al 15 de septiembre estará abierto el periodo para elegir tu plan de gastos médicos. Puedes hacer el trámite desde el portal interno de empleados, entrando con tu usuario habitual.",
    "Te recordamos que el libro Redes Neuronales y Aprendizaje Profundo vence el 20 de agosto. Puedes renovarlo una vez si nadie más lo ha apartado.",
    "Recibimos tu pago de la mensualidad y adjuntamos el recibo correspondiente. Tu membresía queda vigente hasta el 10 de septiembre.",
    "Tu itinerario: vuelo 448, salida a las 07:20 de la Ciudad de México y llegada a las 09:05 a Guadalajara. Te recomendamos documentar con dos horas de anticipación.",
    "Confirmación de reservación: habitación sencilla, entrada el 3 de octubre y salida el 6. La cancelación es gratuita hasta 48 horas antes de la llegada.",
    "Publicamos la versión 2.4. Incluye exportación a CSV, corrige la fuga de memoria del módulo de reportes y mejora el tiempo de carga inicial. Las notas completas están en http://repo-interno.example/proyecto/releases/2.4",
    "Subí los cambios de la rama de autenticación en http://repo-interno.example/proyecto/pr/214. Son unos doscientos renglones, cuando puedas les das una revisada y me dices si te parece bien el manejo de errores.",
    "Mantenimiento programado: el servicio estará fuera de línea el domingo de dos a cinco de la mañana. Guarden su trabajo antes del sábado por la noche.",
    "Gracias por postularte. Nos gustaría agendar una entrevista técnica de una hora la próxima semana. ¿Qué día y qué horario te acomodan mejor?",
    "Los resultados de la convocatoria de becas ya están publicados. Consulta el listado con tu número de folio en el portal escolar al que entras normalmente. El trámite no tiene ningún costo.",
    "Tu recibo de agua del bimestre ya está disponible por $412.00, con vencimiento el 28 de agosto. Puedes pagarlo en http://servicios-agua.example/pago-en-linea, en el módulo de atención o en tiendas autorizadas.",
    "Tu póliza de auto vence el 30 de septiembre. Si deseas renovarla con las mismas coberturas, responde a este correo o márcale directamente a tu agente.",
    "Tu paquete fue entregado hoy a las 11:42 y lo recibió una persona identificada como J. Ramírez. Gracias por comprar con nosotros.",
    "Activaste la verificación en dos pasos en tu cuenta. Si no fuiste tú, entra a la aplicación y revisa tus dispositivos conectados. Este correo es solo informativo.",
    "Te damos la bienvenida. Tu cuenta quedó creada con este correo. En http://plataforma-orion.example/ayuda/inicio encontrarás una guía rápida para empezar.",
    "Tu pedido salió del almacén. El número de rastreo es 993027118 y llega en un plazo de tres a cinco días hábiles.",
    "Procesamos tu devolución. El reembolso de $749.00 se verá reflejado en tu tarjeta en un plazo de cinco a diez días hábiles, según los tiempos de tu banco.",
    "¿Cómo estuvo la atención que recibiste? Nos ayudaría mucho que la califiques del 1 al 5 respondiendo a este correo. No te pedimos ningún dato adicional.",
    "Te esperamos mañana en el seminario en línea sobre modelos de lenguaje. Empieza a las 17:00 y dura hora y media. La liga de acceso es http://seminarios-cic.example/sala/ml-2026, la misma que te enviamos al registrarte.",
    "Como acordamos en la llamada, te envío el contrato de servicios para tu firma. Si prefieres revisarlo antes en un PDF sin firmar, dime y te lo mando. Cualquier duda me marcas al número de siempre.",
    "Ya se publicaron las calificaciones del segundo parcial. Si tienen alguna aclaración pueden verme en asesoría el martes de 12 a 14 horas.",
    "Invitación al encuentro de egresados el 25 de octubre en el auditorio. Habrá mesas de trabajo por generación y un brindis al final. Confirmen su asistencia en http://egresados-cic.example/encuentro2026 o por este medio.",
    "Gracias por participar como voluntario en la jornada del sábado. Entre todos armamos 320 despensas. Dejamos algunas fotos del evento en http://fundacion-raices.example/galeria/jornada-agosto",
    "Les recuerdo que el lunes es día feriado y no habrá actividades. Regresamos el martes con el horario normal.",
    "Resumen de la semana: cerramos la integración con el módulo de pagos y quedan pendientes las pruebas de carga y la documentación. Vamos dos días adelantados respecto al plan.",
    "Notas de la retrospectiva: funcionó bien dividir las tareas en partes más chicas, hay que mejorar la comunicación con el área de diseño y vamos a probar limitar el trabajo en curso a tres tareas.",
    "Te compartí la carpeta con los datos del experimento en la unidad del laboratorio. Ya tienes permiso con tu cuenta institucional, entra como siempre desde el portal y no por una liga externa.",
    "Vacantes de esta semana: dos plazas de analista de datos y una de ingeniero de aprendizaje automático. El detalle está en http://bolsa-trabajo.example/vacantes, si conoces a alguien interesado compárteselo.",
    "En el episodio de esta semana platicamos sobre mecanismos de atención con una investigadora del área. Dura cuarenta minutos y ya está en http://podcast-datos.example/episodios/58",
    "Tu nueva tarjeta está lista y llegará a tu domicilio en un plazo de siete a diez días hábiles. Cuando la recibas, actívala desde la aplicación.",
    "Se realizó el depósito de tu nómina correspondiente a la primera quincena de agosto. Tu recibo está disponible en http://portal-empleados.example/nomina",
    "El permiso de estacionamiento vence el 31 de agosto. Para renovarlo, pasa a la oficina de servicios generales con tu credencial y la tarjeta de circulación.",
    "Tu equipo nuevo ya llegó. Pasa a recogerlo al área de sistemas en horario de oficina y trae tu credencial. La migración de archivos la hacemos contigo en ese momento.",
    "Recordatorio: la fecha límite para entregar la comprobación de gastos del viaje es el viernes 22. Después de esa fecha el sistema ya no acepta comprobantes del periodo.",
    "Concluiste el curso de fundamentos de estadística. Tu constancia con folio 2026-0884 ya está disponible en http://cursos-linea.example/perfil/constancias",
]

if __name__ == "__main__":
    destino = sys.argv[1] if len(sys.argv) > 1 else "data/es_test.csv"
    assert len(PHISHING) == 50, len(PHISHING)
    assert len(LEGITIMOS) == 50, len(LEGITIMOS)
    assert len(set(PHISHING) | set(LEGITIMOS)) == 100, "hay textos repetidos"

    df = pd.DataFrame({
        "texto": PHISHING + LEGITIMOS,
        "etiqueta": [1] * 50 + [0] * 50,
    })
    df.to_csv(destino, index=False, encoding="utf-8")
    print(f"Escrito {destino}: {len(df)} correos, phishing {df.etiqueta.mean():.0%}")
    print("Largo medio (caracteres): phishing "
          f"{df[df.etiqueta == 1].texto.str.len().mean():.0f} | "
          f"legitimos {df[df.etiqueta == 0].texto.str.len().mean():.0f}")
